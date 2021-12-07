import {
  Component,
  Input,
  OnInit,
  SimpleChanges,
  OnChanges,
} from '@angular/core';

@Component({
  selector: 'app-image-output',
  templateUrl: './image-output.component.html',
  styleUrls: ['./image-output.component.css'],
})
export class ImageOutputComponent implements OnInit, OnChanges {
  @Input() imgSrc: [] = [];

  finalUrl = '';
  isImg = false;
  imageArray = [];
  imageObject = {};
  suffix = '../../../assets';
  finalTemp = '';

  constructor() {}

  ngOnChanges(changes: SimpleChanges) {
    let tempObject;
    for (const propName of Object.keys(changes)) {
      const chng = changes[propName];
      const cur = JSON.parse(JSON.stringify(chng.currentValue));
      tempObject = cur;
    }
    console.log(tempObject);
    const Y = 'data';
    const X = tempObject[0];
    let Z = X.split(Y).pop();
    Z = Z.replace(/\\/g, '/');
    console.log('This is the final data url:', Z);
    this.finalUrl = Z;

    for (let current of tempObject) {
      const tempImageObject = {};
      current = this.suffix + current.split(Y).pop().replace(/\\/g, '/');
      console.log(current);
      this.imageArray.push(current);
    }
    console.log(this.imageArray);
    this.finalTemp = this.imageArray[0];
    this.isImg = true;
  }

  ngOnInit(): void {}
}
