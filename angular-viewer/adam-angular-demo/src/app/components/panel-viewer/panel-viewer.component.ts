import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';

@Component({
  selector: 'app-panel-viewer',
  templateUrl: './panel-viewer.component.html',
  styleUrls: ['./panel-viewer.component.css'],
})
export class PanelViewerComponent implements OnChanges {
  @Input() differencesObject;

  similarities = [];
  objectOne;
  objectTwo;
  objectOneArray = [];
  objectTwoArray = [];
  objectsArray = [];
  submit = false;
  constructor() {}

  ngOnChanges(changes: SimpleChanges): void {
    let tempObject;
    for (const propName of Object.keys(changes)) {
      const chng = changes[propName];
      if (typeof chng.currentValue !== 'undefined') {
        const cur = JSON.parse(JSON.stringify(chng.currentValue));
        tempObject = cur;
      }
    }

    if (typeof tempObject === 'undefined') {
      this.clearAppPanel();
    } else {
      this.objectsArray = [];
      for (const key in tempObject) {
        if (key === 'similarities') {
          continue;
        }
        this.objectsArray.push(key);
      }

      this.objectOne = this.objectsArray[0];
      this.objectTwo = this.objectsArray[1];

      this.objectOneArray = tempObject[this.objectOne];
      this.objectTwoArray = tempObject[this.objectTwo];
      this.similarities = tempObject.similarities;
      this.submit = true;
    }
  }

  clearAppPanel(): void {
    this.similarities = [];
    this.objectOne = null;
    this.objectTwo = null;
    this.objectOneArray = [];
    this.objectTwoArray = [];
    this.objectsArray = [];
    this.submit = false;
  }
}
