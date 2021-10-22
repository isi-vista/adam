import { Component, Input, OnInit } from '@angular/core';

@Component({
  selector: 'app-image-output',
  templateUrl: './image-output.component.html',
  styleUrls: ['./image-output.component.css']
})
export class ImageOutputComponent implements OnInit {


  @Input() img_src:string;
  
  constructor() { }

  ngOnInit(): void {
  }

}
