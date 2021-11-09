import {
  Component,
  OnInit,
  Input,
  SimpleChange,
  SimpleChanges,
  OnChanges,
} from '@angular/core';
import { AdamService } from 'src/app/services/adam.service';
import { MainObject } from 'src/app/classes/main-object';
import { Features } from 'src/app/classes/features';
import { SubObject } from 'src/app/classes/sub-object';

@Component({
  selector: 'app-object-results',
  templateUrl: './object-results.component.html',
  styleUrls: ['./object-results.component.css'],
})
export class ObjectResultsComponent implements OnInit, OnChanges {
  @Input() outputObject;

  resultObject;
  sceneNumber;
  subObjects;

  result = new MainObject();
  resultArray = new Array<MainObject>();
  isObject = false;

  constructor(private getResponseData: AdamService) {}

  ngOnChanges(changes: SimpleChanges) {
    let tempObject;
    for (const propName of Object.keys(changes)) {
      const chng = changes[propName];
      const cur = JSON.parse(JSON.stringify(chng.currentValue));
      tempObject = cur;
    }
    console.log(tempObject);

    for (const entry of tempObject.main) {
      const tempMain = new MainObject();
      tempMain.text = entry.text;
      tempMain.confidence = entry.confidence;
      tempMain.features = new Array<Features>();
      tempMain.subObject = new Array<SubObject>();
      entry.features.forEach((element) => {
        const feat = new Features();
        feat.name = element.name;
        tempMain.features.push(feat);
      });
      if (entry.hasOwnProperty('sub_objects')) {
        entry.sub_objects.forEach((element) => {
          const subobject = new SubObject();
          subobject.confidence = element.confidence;
          subobject.text = element.text;
          subobject.features = new Array<Features>();
          element.features.forEach((feature) => {
            const feat = new Features();
            feat.name = feature.name;
            subobject.features.push(feat);
          });
          tempMain.subObject.push(subobject);
        });
      }
      this.resultArray.push(tempMain);
    }

    console.log(this.resultArray);

    this.isObject = true;
  }

  ngOnInit(): void {}
}
