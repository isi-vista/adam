import { Features } from './features';

export class LinguisticOutput {
  id: number;
  text: string;
  confidence: number;
  type: string;
  features: Features[];
  sub_objects: LinguisticOutput[];
  raw_text?: string;
  slot_alignment_to_confidence?: Map<string, Map<string, number>>;
}
