require("dotenv").config();
const express = require("express");
const axios = require("axios");
const path = require("path");
const { GoogleGenerativeAI } = require("@google/generative-ai");

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

const NAVER_CLIENT_ID = (process.env.NAVER_CLIENT_ID || "").trim();
const NAVER_CLIENT_SECRET = (process.env.NAVER_CLIENT_SECRET || "").trim();
const genAI = new GoogleGenerativeAI((process.env.GEMINI_API_KEY || "").trim());

const STOP_WORDS = new Set(["무료배송", "공식", "정품", "행사", "쿠폰"]);

function getNaverHeaders() {
  return {
    "X-Naver-Client-Id": NAVER_CLIENT_ID,
    "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
  };
}

// --- 네이버 API 헬퍼 ---

async function searchNaverShopping(query, display = 100) {
  const res = await axios.get("https://openapi.naver.com/v1/search/shop.json", {
    params: { query, display, start: 1, sort: "sim" },
    headers: getNaverHeaders(),
  });
  return res.data.items || [];
}

async function getShoppingTotal(query) {
  const res = await axios.get("https://openapi.naver.com/v1/search/shop.json", {
    params: { query, display: 1, start: 1, sort: "sim" },
    headers: getNaverHeaders(),
  });
  return res.data.total || 0;
}

async function getSearchTrends(keywords) {
  const now = new Date();
  const endDate = now.toISOString().slice(0, 10);
  const past = new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000);
  const startDate = past.toISOString().slice(0, 10);
  const res = await axios.post(
    "https://openapi.naver.com/v1/datalab/search",
    {
      startDate,
      endDate,
      timeUnit: "month",
      keywordGroups: keywords.map((k) => ({ groupName: k, keywords: [k] })),
    },
    { headers: { ...getNaverHeaders(), "Content-Type": "application/json" } }
  );
  return res.data.results || [];
}

// --- 분석 유틸 ---

function inferMainCategory(items) {
  const freq = {};
  for (const item of items) {
    const parts = [item.category1, item.category2, item.category3, item.category4].filter(Boolean);
    const key = parts.join(">");
    freq[key] = (freq[key] || 0) + 1;
  }
  let maxKey = "", maxCount = 0;
  for (const [key, count] of Object.entries(freq)) {
    if (count > maxCount) { maxCount = count; maxKey = key; }
  }
  const split = maxKey.split(">");
  return { category1: split[0] || "", category2: split[1] || "", category3: split[2] || "", category4: split[3] || "" };
}

function stripHtml(str) {
  return str.replace(/<[^>]*>/g, "");
}

function buildKeywordStats(items) {
  const freq = {};
  for (const item of items) {
    const words = stripHtml(item.title).split(/\s+/);
    for (const w of words) {
      if (w.length < 2 || STOP_WORDS.has(w)) continue;
      freq[w] = (freq[w] || 0) + 1;
    }
  }
  return Object.entries(freq).sort((a, b) => b[1] - a[1]).slice(0, 50);
}

// --- 핵심 분석 함수 ---

async function analyzeProductName(productName) {
  const firstItems = await searchNaverShopping(productName);
  const mainCategory = inferMainCategory(firstItems);
  const categoryPrefix = (mainCategory.category4 || mainCategory.category3 || "").trim();
  const secondQuery = `${categoryPrefix} ${productName}`.trim();
  const secondItems = await searchNaverShopping(secondQuery);
  const topTitles = secondItems.map((item) => stripHtml(item.title));
  const keywordStats = buildKeywordStats(secondItems);
  return { inputProductName: productName, mainCategory, topTitles, keywordStats };
}

// --- 키워드 메트릭 수집 ---

async function getKeywordMetrics(keywordStats, topN = 15) {
  const topKeywords = keywordStats.slice(0, topN).map(([kw]) => kw);

  // 상품 수 (경쟁도 지표) - 병렬 호출
  const totals = await Promise.all(
    topKeywords.map((kw) => getShoppingTotal(kw).catch(() => 0))
  );

  // 트렌드 데이터 - 5개씩 배치 (DataLab API 제한)
  const trendMap = {};
  try {
    for (let i = 0; i < topKeywords.length; i += 5) {
      const batch = topKeywords.slice(i, i + 5);
      const results = await getSearchTrends(batch);
      for (const r of results) {
        const data = r.data || [];
        if (data.length >= 6) {
          const recent = data.slice(-3).reduce((s, d) => s + d.ratio, 0) / 3;
          const earlier = data.slice(0, 3).reduce((s, d) => s + d.ratio, 0) / 3;
          trendMap[r.title] = {
            direction: recent > earlier * 1.15 ? "rising" : recent < earlier * 0.85 ? "declining" : "stable",
            latestRatio: Math.round(data[data.length - 1]?.ratio || 0),
          };
        }
      }
    }
  } catch (e) {
    // DataLab API 미활성화 시 무시
  }

  return topKeywords.map((kw, i) => {
    const total = totals[i];
    let competitionLevel;
    if (total < 5000) competitionLevel = "low";
    else if (total < 30000) competitionLevel = "medium";
    else if (total < 100000) competitionLevel = "high";
    else competitionLevel = "very_high";

    const trend = trendMap[kw] || { direction: "unknown", latestRatio: 0 };
    return {
      keyword: kw,
      totalProducts: total,
      competitionLevel,
      trend: trend.direction,
      trendRatio: trend.latestRatio,
      frequencyInTitles: keywordStats.find(([k]) => k === kw)?.[1] || 0,
    };
  });
}

// --- Gemini 호출 헬퍼 ---

function stripCodeFence(text) {
  return text.replace(/^```(?:json|markdown)?\s*/i, "").replace(/\s*```\s*$/i, "").trim();
}

async function callGemini(prompt, maxTokens = 2048) {
  const model = genAI.getGenerativeModel({
    model: "gemini-2.5-flash",
    generationConfig: {
      maxOutputTokens: maxTokens,
      thinkingConfig: { thinkingBudget: 0 },
    },
  });
  const result = await model.generateContent(prompt);
  return stripCodeFence(result.response.text());
}

async function generateSuggestedTitles(analysis) {
  const prompt = `너는 네이버 스마트스토어 상품명 작성 전문가다.

아래 데이터를 참고하여 네이버 쇼핑 SEO에 최적화된 추천 상품명 10개를 만들어라.

[입력 상품명] ${analysis.inputProductName}
[대표 카테고리] ${JSON.stringify(analysis.mainCategory)}
[상위 노출 상품명 (상위 20개)] ${JSON.stringify(analysis.topTitles.slice(0, 20))}
[키워드 통계 (상위 20개)] ${JSON.stringify(analysis.keywordStats.slice(0, 20))}

[규칙]
1. 핵심 키워드를 앞으로 배치하여 네이버 쇼핑 SEO에 유리하게 작성.
2. keywordStats 상위 키워드를 중심으로 7개 이하의 단어로 구성.
3. 40~50자 이내, 읽기 쉬운 공백/구분기호 사용.
4. "무료배송", "행사", "쿠폰"은 사용하지 말 것.
5. 총 10개의 상품명을 만들어라.

[출력 형식]
반드시 JSON 배열만 반환. 다른 텍스트 없이:
["상품명1","상품명2",...,"상품명10"]`;

  const raw = await callGemini(prompt);
  try {
    return JSON.parse(raw);
  } catch {
    const match = raw.match(/\[[\s\S]*\]/);
    return match ? JSON.parse(match[0]) : [raw];
  }
}

async function generateReport(analysis, keywordMetrics, suggestedTitles) {
  const prompt = `너는 네이버 스마트스토어 상품명·키워드 전략을 강의하는 강사다.
아래 JSON은 특정 상품에 대해 수집한 분석 결과다.
이 데이터를 기반으로, "키워드는 나열하고, 우리 제품과 관련 없는 것은 버리고, 남은 키워드를 소비자 입장에서 보기 좋게 배열한 뒤, 작은 키워드 시장부터 계단식으로 확장한다"라는 철학에 맞는 전문 리포트를 작성해줘.

${JSON.stringify({ analysis: { ...analysis, keywordMetrics }, suggestedTitles }, null, 2)}

리포트는 다음 목차를 반드시 포함해야 한다.

## 1. 상품 개요 및 시장 포지셔닝
- inputProductName, mainCategory, 상위 topTitles를 바탕으로 "이 상품이 어떤 시장에 있는지", "경쟁이 어떤 구조인지"를 요약해라.
- "상품명만 바꿔서 되는 게임이 아니라, 팔리는 제품 + 시장 이해가 먼저"라는 메시지를 짧게 언급해라.

## 2. 키워드 구조 분석 (나열 단계)
- keywordStats와 keywordMetrics를 사용해, 키워드를 아래 네 그룹으로 나눠 표 형식이나 리스트로 설명해라.
  - **메인 키워드** (예: 복숭아, 욕실청소세제 등)
  - **품종/브랜드/속성 키워드** (예: 신비, 황도, 프리미엄, 저자극 등)
  - **형태/질감/용도 키워드** (예: 딱딱이, 아삭, 대용량, 원룸용 등)
  - **지역/차별화 키워드** (예: 함양, 조치원, 국내산, 자사브랜드 등)
- 각 그룹별로 "이 시장에서 특히 의미 있는 포인트"를 한두 문장으로 정리해라.

## 3. 우리 제품 연관성 필터링 (제거 단계)
- keywordStats와 keywordMetrics를 보면서, "검색량은 있지만 현재 제품 스펙과 연관성이 낮아 보이는 키워드"를 3~7개 정도 골라 "욕심 키워드"로 지적해라.
- 왜 위험한지를 구체적으로 설명하고, 상품명에서는 제외하라고 조언해라.

## 4. 상품명 설계 전략 (배열 단계)
- 이 상품에 대해 추천하는 기본 구조를 한 줄로 정의해라. 예: [주체 키워드] + [형태/효과] + [품종/스펙]
- suggestedTitles 10개를 유형별로 나눠서 요약 평가해라.
- 강의 철학 기준에서 "가장 추천하는 상품명 2~3개"를 골라서, 왜 좋은지(연관성, 소비자 가독성, 계단 전략에 유리한지)를 구체적으로 설명해라.

## 5. 진입 키워드 → 확장 키워드 계단 전략
- keywordMetrics의 totalProducts(경쟁 상품 수)와 competitionLevel을 기준으로:
  - **진입용 키워드** (경쟁 낮음, 진입 쉬움)
  - **중간 단계 키워드**
  - **메인/대형 키워드**
  세 구간으로 나눠라.
- 각 구간별로 "어떤 순서로 공략해야 하는지", "어떤 콘텐츠/광고/리뷰 액션이 필요한지"를 계단식 전략으로 서술해라.
- '770 → 2000 → 5000 → 20만' 같은 흐름을 이 상품의 실제 숫자에 맞게 비유해서 설명해라.

## 6. 리스크 및 추가 개선 제안
- 현재 상품명/키워드 전략만 봤을 때 예상되는 한계를 냉정하게 지적해라.
- "상품명만 바꿔서 매출이 보장되진 않는다"는 메시지를, 이 상품에 맞는 구체적인 조언과 함께 정리해라.

문체와 형식:
- 마크다운 헤더와 리스트를 적극 활용해서, 바로 블로그/노션에 붙여넣어도 읽기 편하게 작성해라.
- 말투는 실제 강의하듯, 초보 셀러도 이해할 수 있게 쉽게 설명하되, 방향성은 단호하고 직설적으로.
- 불필요한 예의 표현은 줄이고, actionable insight에 집중해라.`;

  return await callGemini(prompt, 8000);
}

// --- 라우트 1: 분석만 ---

app.post("/generate-titles", async (req, res) => {
  try {
    const { productName } = req.body;
    if (!productName) return res.status(400).json({ error: "productName is required" });
    const result = await analyzeProductName(productName);
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// --- 라우트 2: 분석 + Claude 추천 ---

app.post("/generate-titles-and-suggest", async (req, res) => {
  try {
    const { productName } = req.body;
    if (!productName) return res.status(400).json({ error: "productName is required" });
    const analysis = await analyzeProductName(productName);
    const suggestedTitles = await generateSuggestedTitles(analysis);
    res.json({ analysis, suggestedTitles });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// --- 라우트 3: 전체 분석 + 메트릭 + 추천 + 전략 리포트 ---

app.post("/generate-report", async (req, res) => {
  try {
    const { productName } = req.body;
    if (!productName) return res.status(400).json({ error: "productName is required" });

    // 기본 분석
    const analysis = await analyzeProductName(productName);

    // 키워드 메트릭 + 추천 상품명 병렬 수집
    const [keywordMetrics, suggestedTitles] = await Promise.all([
      getKeywordMetrics(analysis.keywordStats),
      generateSuggestedTitles(analysis),
    ]);

    // 전략 리포트 생성
    const report = await generateReport(analysis, keywordMetrics, suggestedTitles);

    res.json({
      analysis: { ...analysis, keywordMetrics },
      suggestedTitles,
      report,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3000;
if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
}

module.exports = app;
