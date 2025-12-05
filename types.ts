export enum ProjectType {
  RESIDENTIAL = 'Residential',
  COMMERCIAL = 'Commercial',
  INDUSTRIAL = 'Industrial',
  RENOVATION = 'Renovation'
}

export enum QualityLevel {
  ECONOMY = 'Economy',
  STANDARD = 'Standard',
  PREMIUM = 'Premium'
}

export interface ProjectInputs {
  type: ProjectType;
  quality: QualityLevel;
  location: string; // "City, State"
  sizeSqFt: number;
  budgetLimit: number;
  timelineMonths: number;
  manpower: number;
}

export interface CostItem {
  category: string;
  cost: number;
  description: string;
}

export interface CashFlowMonth {
  month: number;
  amount: number;
  phase: string;
}

export interface RiskItem {
  risk: string;
  impact: 'Low' | 'Medium' | 'High';
  mitigation: string;
}

export interface EstimationResult {
  currencySymbol: string;
  totalEstimatedCost: number;
  breakdown: CostItem[];
  cashflow: CashFlowMonth[];
  risks: RiskItem[];
  confidenceScore: number;
  confidenceReason: string;
  efficiencyTips: string[];
  summary: string;
}

export interface FeasibilityResult {
  isValid: boolean;
  score: number;
  budgetVerdict: 'Realistic' | 'Insufficient' | 'Excessive'; 
  issues: string[];
  suggestions: string[];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'model';
  text: string;
  timestamp: Date;
}
