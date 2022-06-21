import { Affordances } from './affordances';
import { Features } from './features';

export class LinguisticOutput {
  id: number;
  text: string;
  confidence: number;
  type: string;
  features: Features[];
  affordances: Affordances[];
  sub_objects: LinguisticOutput[];
  raw_text?: string;
  slot_alignment_to_confidence?: Map<string, Map<string, number>>;
}
