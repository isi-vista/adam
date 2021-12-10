import {
  Component,
  OnInit,
  Input,
  SimpleChanges,
  OnChanges,
} from '@angular/core';
import { TouchSequence } from 'selenium-webdriver';

@Component({
  selector: 'app-panel-viewer',
  templateUrl: './panel-viewer.component.html',
  styleUrls: ['./panel-viewer.component.css'],
})
export class PanelViewerComponent implements OnInit, OnChanges {
  @Input() differencesObject;

  similarities = [];
  objectOne;
  objectTwo;
  objectOneArray = [];
  objectTwoArray = [];
  objectsArray = [];
  submit = false;
  constructor() {}

  ngOnChanges(changes: SimpleChanges) {
    let tempObject;
    for (const propName of Object.keys(changes)) {
      const chng = changes[propName];
      const cur = JSON.parse(JSON.stringify(chng.currentValue));
      tempObject = cur;
    }
    console.log('Differences change:', tempObject);

    console.log(Object.keys(tempObject));
    for (const key in tempObject) {
      if (key === 'similarities') {
        continue;
      }
      this.objectsArray.push(key);
    }
    console.log(this.objectsArray);

    this.objectOne = this.objectsArray[0];
    this.objectTwo = this.objectsArray[1];

    this.objectOneArray = tempObject[this.objectOne];
    this.objectTwoArray = tempObject[this.objectTwo];
    this.similarities = tempObject.similarities;

    console.log(this.objectOneArray);
    this.submit = true;
  }

  ngOnInit(): void {}
}
