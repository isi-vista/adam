import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { Features } from '../../classes/features';
import { MainObject } from '../../classes/main-object';
import { SubObject } from '../../classes/sub-object';

@Component({
  selector: 'app-object-results',
  templateUrl: './object-results.component.html',
  styleUrls: ['./object-results.component.css'],
})
export class ObjectResultsComponent implements OnChanges {
  @Input() outputObject;

  resultArray: Array<MainObject> = [];
  isObject = false;

  constructor() {}

  ngOnChanges(changes: SimpleChanges): void {
    let tempObject;
    for (const propName of Object.keys(changes)) {
      const chng = changes[propName];
      const cur = JSON.parse(JSON.stringify(chng.currentValue));
      tempObject = cur;
    }

    this.resultArray = [];

    for (const entry of tempObject.main) {
      const tempMain = new MainObject();
      tempMain.text = entry.text;
      tempMain.confidence = entry.confidence;
      tempMain.features = new Array<Features>();
      tempMain.subObject = new Array<SubObject>();
      entry.features.forEach((element) => {
        const feat = new Features();
        feat.name = element;
        tempMain.features.push(feat);
      });
      if (entry.sub_objects) {
        entry.sub_objects.forEach((element) => {
          const subobject = new SubObject();
          subobject.confidence = element.confidence;
          subobject.text = element.text;
          subobject.features = new Array<Features>();
          element.features.forEach((feature) => {
            const feat = new Features();
            feat.name = feature;
            subobject.features.push(feat);
          });
          tempMain.subObject.push(subobject);
        });
      }
      this.resultArray.push(tempMain);
    }

    this.isObject = true;
  }
}
