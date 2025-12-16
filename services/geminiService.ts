import OpenAI from "openai";
import { ProjectInputs, EstimationResult, FeasibilityResult } from '../types';

// Initialize OpenAI client pointing to OpenRouter
const client = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: import.meta.env.VITE_OPENROUTER_API_KEY,
  dangerouslyAllowBrowser: true // Client-side usage
});

// Using a reliable and widely available free model from OpenRouter.
// The 'google/gemini-flash-1.5' model can sometimes return a 404 if free providers are unavailable
// or if account privacy settings do not allow for models that train on data.
// Mistral-7B-Instruct is a stable and capable alternative.
const MODEL = "mistralai/mistral-7b-instruct:free";

// Helper to clean JSON if model returns markdown
const cleanJson = (text: string): string => {
  const match = text.match(/```json\n([\s\S]*?)\n```/) || text.match(/```([\s\S]*?)```/);
  return match ? match[1] : text;
};

// Retry helper with exponential backoff for rate limits
const retryWithBackoff = async <T>(
  fn: () => Promise<T>,
  maxRetries: number = 5,
  baseDelay: number = 2000
): Promise<T> => {
  let lastError: any;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error: any) {
      lastError = error;

      // Check if it's a rate limit error (429)
      if (error?.status === 429 || error?.code === 'rate_limit_exceeded' || error?.message?.includes('rate limit')) {
        if (attempt < maxRetries) {
          const delay = baseDelay * Math.pow(2, attempt); // Exponential backoff: 2s, 4s, 8s, 16s, 32s
          console.warn(`Rate limit hit, retrying in ${delay}ms... (attempt ${attempt + 1}/${maxRetries + 1})`);
          await new Promise(resolve => setTimeout(resolve, delay));
          continue;
        }
      }

      // For other errors, throw immediately
      throw error;
    }
  }

  throw lastError;
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

      2. **Physical/Labor Feasibility (CRITICAL - Can manpower complete work in timeline?)**:
         - Determine the standard working days per month in ${inputs.location} (typically 25-26 working days/month).
         - Estimate the **Total Man-Days Required** to complete a ${inputs.sizeSqFt} sq ft ${inputs.type} project of ${inputs.quality} quality.
            - Use realistic productivity rates based on project type:
              * Residential: 15-20 sq ft/day per worker
              * Commercial: 10-15 sq ft/day per worker
              * Industrial: 8-12 sq ft/day per worker
              * Renovation: 12-18 sq ft/day per worker
            - Adjust for quality level: Economy +20% productivity, Premium -15% productivity.
         - Calculate **Available Man-Days** = ${inputs.manpower} workers × ${inputs.timelineMonths} months × 25 working days/month.
         - **TIMELINE FEASIBILITY CHECK**: Compare Required vs Available Man-Days.
         - If Available < Required, calculate how many additional workers needed OR how many extra months required.
         - This determines if the project can realistically be completed with given manpower and timeline.

      OUTPUT: Return ONLY a valid JSON object with this structure:
      {
        "isValid": boolean,
        "budgetVerdict": "Realistic" | "Insufficient" | "Excessive",
        "issues": ["string"],
        "suggestions": ["string"]
      }

      CRITICAL REQUIREMENTS FOR MANPOWER/TIMELINE ANALYSIS:
      - If manpower/timeline is insufficient, you MUST add a HIGH-PRIORITY issue as the FIRST item: "TIMELINE FEASIBILITY: With current manpower, this project requires X months minimum OR you need Y additional workers for the planned timeline."
      - Calculate and include specific numbers: Required Man-Days, Available Man-Days, and exact recommendations.
      - Make manpower/timeline issues the FIRST item in the issues array when they exist.
      - Provide clear, actionable suggestions: "Increase manpower to X workers" OR "Extend timeline to Y months" OR "Both options available".
      - If timeline is feasible, explicitly state: "Manpower sufficient: X workers can complete in Y months."
    `;

    const response = await retryWithBackoff(() =>
      client.chat.completions.create({
        model: MODEL,
        messages: [{ role: "user", content: prompt }],
        response_format: { type: "json_object" }, // This is supported by many models for JSON output
        stream: false
      })
    );

    const content = response.choices?.[0]?.message?.content || "{}";
    const data = JSON.parse(cleanJson(content));

    return {
      isValid: !!data.isValid,
      budgetVerdict: data.budgetVerdict || 'Insufficient',
      issues: data.issues || [],
      suggestions: data.suggestions || []
    };
  } catch (error) {
    console.error("Feasibility check failed:", error);
    return {
      isValid: false,
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
         - Research and use ACCURATE local daily/monthly wage rates for construction workers in ${inputs.location} for the current year.
         - Calculate this specifically for ${inputs.manpower} workers over ${inputs.timelineMonths} months.
         - Include relevant categories for ${inputs.type} projects: Materials, Equipment Rental, Permits & Licenses, Site Preparation, Foundation, Structural Work, Electrical & Plumbing, Finishing, Contingency (5-10%), Transportation, Insurance, and any location-specific costs.
         - Ensure all costs reflect current market rates in ${inputs.location}.

      3. **Cashflow (Crucial)**:
         - The 'cashflow' array MUST have EXACTLY ${inputs.timelineMonths} entries.
         - It must range from Month 1 to Month ${inputs.timelineMonths}.
         - Do not generate 12 months if the timeline is ${inputs.timelineMonths}.

      4. **Manpower Feasibility Analysis (CRITICAL for Confidence)**:
         - Calculate Total Man-Days Required = Estimate man-days needed for ${inputs.sizeSqFt} sq ft ${inputs.type} project of ${inputs.quality} quality.
         - Calculate Available Man-Days = ${inputs.manpower} workers × ${inputs.timelineMonths} months × 25 working days/month.
         - If Available Man-Days < Required Man-Days, this severely impacts confidence score (reduce by 30-50 points).
         - Include manpower feasibility assessment in 'confidenceReason'.

      5. **Confidence Score Calculation**:
         - Start with base score of 85-95 for good data availability.
         - Reduce by 10-20 points if location data is uncertain.
         - Reduce by 30-50 points if manpower is insufficient for timeline.
         - Reduce by 15-25 points if budget is unrealistic.
         - Ensure score reflects overall reliability including manpower constraints.

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
    const response = await retryWithBackoff(() =>
      client.chat.completions.create({
        model: MODEL,
        messages: [{ role: "user", content: prompt }],
        response_format: { type: "json_object" }
      })
    );

    const content = response.choices?.[0]?.message?.content || "{}";
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
    stream: false
  });

  return response.choices?.[0]?.message?.content || "";
};

// 4. Edit Site Image (Stub - Grok is text only)
export const editSiteImage = async (base64Data: string, prompt: string): Promise<string | null> => {
  console.warn("Image editing is not supported by the current model.");
  return null;
}
