import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';

@Component({
  selector: 'app-image-output',
  templateUrl: './image-output.component.html',
  styleUrls: ['./image-output.component.css'],
})
export class ImageOutputComponent implements OnChanges {
  @Input() imgSrc: [] = [];

  isImg = false;
  imageArray = [];
  imageObject = {};
  suffix = '../../../assets';

  constructor() {}

  ngOnChanges(changes: SimpleChanges): void {
    this.imageArray = [];
    let tempObject;
    for (const propName of Object.keys(changes)) {
      const chng = changes[propName];
      const cur = JSON.parse(JSON.stringify(chng.currentValue));
      tempObject = cur;
    }

    for (const current of tempObject) {
      this.imageArray.push(
        this.suffix + current.split('data').pop().replace(/\\/g, '/')
      );
    }
    this.isImg = true;
  }
}
