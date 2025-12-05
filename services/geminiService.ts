import OpenAI from "openai";
import { ProjectInputs, EstimationResult, FeasibilityResult } from '../types';

// Initialize OpenAI client pointing to OpenRouter
const client = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: import.meta.env.VITE_OPENROUTER_API_KEY,
  dangerouslyAllowBrowser: true // Client-side usage
});

const MODEL = "x-ai/grok-4.1-fast";

// Helper to clean JSON if model returns markdown
const cleanJson = (text: string): string => {
  const match = text.match(/```json\n([\s\S]*?)\n```/) || text.match(/```([\s\S]*?)```/);
  return match ? match[1] : text;
};

// 1. Feasibility Check
export const checkProjectFeasibility = async (inputs: ProjectInputs): Promise<FeasibilityResult> => {
  try {
    const prompt = `
      Act as a construction project manager and cost estimator for ${inputs.location}.
      
      Project Inputs:
      - Type: ${inputs.type}
      - Size: ${inputs.sizeSqFt} sq ft
      - Budget: ${inputs.budgetLimit}
      - Quality: ${inputs.quality}
      - Timeline: ${inputs.timelineMonths} months
      - Manpower: ${inputs.manpower} workers
      
      TASK:
      1. **Financial Feasibility**:
         - Calculate Budget Per Sq Ft = ${inputs.budgetLimit} / ${inputs.sizeSqFt}.
         - Estimate the AVERAGE Market Rate per sq ft for ${inputs.quality} ${inputs.type} construction in ${inputs.location} (Total Project Cost including materials, labor, finishes).
         - Compare:
             - If Budget Per Sq Ft < Market Rate * 0.8 => 'Insufficient'
             - If Budget Per Sq Ft > Market Rate * 3.0 => 'Excessive'
             - Otherwise => 'Realistic'

      2. **Physical/Labor Feasibility (CRITICAL)**:
         - Determine the standard/official labor working hours and days per month in ${inputs.location} (e.g., ~25-26 days/month, 8-9 hours/day).
         - Estimate the **Total Man-Days Required** to complete a ${inputs.sizeSqFt} sq ft ${inputs.type} project of ${inputs.quality} quality.
         - Calculate **Available Man-Days** = ${inputs.manpower} workers * ${inputs.timelineMonths} months * [Working Days Per Month].
         - **CHECK**: Is Available Man-Days >= Required Man-Days?
         - If Available < Required, the project is **Physically Impossible** under normal conditions.

      3. **Scoring & Verdict**:
         - If Financial is 'Insufficient' OR Physical is 'Impossible', Score must be < 60.
         - If both are good, Score > 80.
         - If 'Excessive' budget, Score should be high (90+).
      
      OUTPUT: Return ONLY a valid JSON object with this structure:
      {
        "isValid": boolean,
        "score": number,
        "budgetVerdict": "Realistic" | "Insufficient" | "Excessive",
        "issues": ["string"],
        "suggestions": ["string"]
      }
      
      IMPORTANT: If the manpower/timeline is insufficient, you MUST add a specific issue like "Timeline/Manpower mismatch: Need approx X months with Y workers for this size."
    `;

    const response = await client.chat.completions.create({
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" },
      // @ts-ignore - passing extra body for reasoning
      extra_body: { reasoning: { enabled: true } } 
    });

    const content = response.choices[0].message.content || "{}";
    const data = JSON.parse(cleanJson(content));

    return {
      isValid: !!data.isValid,
      score: data.score || 0,
      budgetVerdict: data.budgetVerdict || 'Insufficient',
      issues: data.issues || [],
      suggestions: data.suggestions || []
    };
  } catch (error) {
    console.error("Feasibility check failed:", error);
    return {
      isValid: false,
      score: 0,
      budgetVerdict: 'Insufficient',
      issues: ["Service unavailable or parse error"],
      suggestions: []
    };
  }
};

// 2. Detailed Estimation
export const generateConstructionEstimate = async (inputs: ProjectInputs): Promise<EstimationResult> => {
  const prompt = `
      Act as an expert chartered surveyor. Estimate construction costs for:
      
      - Type: ${inputs.type}
      - Quality: ${inputs.quality}
      - Location: ${inputs.location}
      - Size: ${inputs.sizeSqFt} sq ft
      - Budget Limit: ${inputs.budgetLimit}
      - Project Timeline: ${inputs.timelineMonths} months
      - Manpower/Workers: ${inputs.manpower} people
      
      REQUIREMENTS:
      1. **Total Cost**: Ensure 'totalEstimatedCost' is a realistic market value. Do not blindly match the budget.
      
      2. **Cost Breakdown (Crucial)**: 
         - You MUST include a dedicated line item in 'breakdown' for "Labor & Wages".
         - Calculate this specifically for ${inputs.manpower} workers over ${inputs.timelineMonths} months based on local daily/monthly wage rates in ${inputs.location}.
         - Include other standard categories (Materials, Permits, etc).
      
      3. **Cashflow (Crucial)**:
         - The 'cashflow' array MUST have EXACTLY ${inputs.timelineMonths} entries.
         - It must range from Month 1 to Month ${inputs.timelineMonths}.
         - Do not generate 12 months if the timeline is ${inputs.timelineMonths}.
      
      OUTPUT: Return ONLY a valid JSON object with this structure:
      {
        "currencySymbol": "string",
        "totalEstimatedCost": number,
        "breakdown": [ { "category": "string", "cost": number, "description": "string" } ],
        "cashflow": [ { "month": number, "amount": number, "phase": "string" } ],
        "risks": [ { "risk": "string", "impact": "Low"|"Medium"|"High", "mitigation": "string" } ],
        "confidenceScore": number,
        "confidenceReason": "string",
        "efficiencyTips": ["string"],
        "summary": "string"
      }
  `;

  try {
    const response = await client.chat.completions.create({
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" },
      // @ts-ignore
      extra_body: { reasoning: { enabled: true } }
    });

    const content = response.choices[0].message.content || "{}";
    return JSON.parse(cleanJson(content)) as EstimationResult;
  } catch (error) {
    console.error("Estimation failed:", error);
    throw error;
  }
};

// 3. Chat
export const sendChatMessage = async (history: {role: string, parts: {text: string}[]}[], message: string) => {
  // Convert Gemini history format to OpenAI format
  const messages = history.map(h => ({
    role: h.role === 'model' ? 'assistant' : 'user',
    content: h.parts[0].text
  })) as any[];

  messages.push({ role: 'user', content: message });

  const response = await client.chat.completions.create({
    model: MODEL,
    messages: messages,
    // @ts-ignore
    extra_body: { reasoning: { enabled: true } }
  });

  return response.choices[0].message.content || "";
};

// 4. Edit Site Image (Stub - Grok is text only)
export const editSiteImage = async (base64Data: string, prompt: string): Promise<string | null> => {
  console.warn("Image editing is not supported by the current model.");
  return null;
};
