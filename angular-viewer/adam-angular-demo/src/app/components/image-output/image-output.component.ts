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

  isImg = false;
  imageArray = [];
  imageObject = {};
  suffix = '../../../assets';

  constructor() {}

  ngOnChanges(changes: SimpleChanges) {
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

  ngOnInit(): void {}
}
