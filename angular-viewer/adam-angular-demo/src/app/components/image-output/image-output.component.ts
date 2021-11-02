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

  constructor() {}

  ngOnChanges(changes: SimpleChanges) {
    let tempObject;
    for (const propName of Object.keys(changes)) {
      const chng = changes[propName];
      const cur = JSON.parse(JSON.stringify(chng.currentValue));
      tempObject = cur;
    }
    console.log(tempObject[0]);
    const Y = 'data';
    const X = tempObject[0];
    let Z = X.split(Y).pop();
    Z = Z.replace(/\\/g, '/');
    console.log('This is the final data url:', Z);
    this.finalUrl = Z;
    this.isImg = true;
  }

  ngOnInit(): void {}
}
