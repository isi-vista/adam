import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { Features } from '../../classes/features';
import { LinguisticOutput } from '../../classes/linguistic-output';

@Component({
  selector: 'app-object-results',
  templateUrl: './object-results.component.html',
  styleUrls: ['./object-results.component.css'],
})
export class ObjectResultsComponent implements OnChanges {
  @Input() outputObject;

  resultArray: Array<LinguisticOutput> = [];
  isObject = false;

  constructor() {}

  private makeLinguisticObject(entry): LinguisticOutput {
    const is_color_re = /#[0-9a-fA-F]{6}/;
    const features: Features[] = [];
    entry.features.forEach((element) => {
      const feat = {
        name: element,
        isColor: is_color_re.test(element),
      };
      features.push(feat);
    });

    const sub_objects: LinguisticOutput[] = [];

    if (entry.sub_objects) {
      entry.sub_objects.forEach((element) => {
        sub_objects.push(this.makeLinguisticObject(element));
      });
    }

    return {
      id: entry.id,
      text: entry.text,
      confidence: entry.confidence,
      type: entry.type,
      features: features,
      sub_objects: sub_objects,
    };
  }

  ngOnChanges(changes: SimpleChanges): void {
    let tempObject;
    for (const propName of Object.keys(changes)) {
      const chng = changes[propName];
      const cur = JSON.parse(JSON.stringify(chng.currentValue));
      tempObject = cur;
    }

    this.resultArray = [];

    for (const entry of tempObject.main) {
      this.resultArray.push(this.makeLinguisticObject(entry));
    }

    this.isObject = true;
  }
}
