import { Component, Input, OnInit, SimpleChanges } from '@angular/core';

@Component({
  selector: 'app-image-output',
  templateUrl: './image-output.component.html',
  styleUrls: ['./image-output.component.css']
})
export class ImageOutputComponent implements OnInit {

  @Input() img_src: []=[];

  final_url = ""
  isImg=false

  constructor() { }

  ngOnChanges(changes: SimpleChanges){
    let tempObject;
    for(const propName in changes){
      const chng = changes[propName]
      const cur = JSON.parse(JSON.stringify(chng.currentValue))
      const prev = JSON.stringify(chng.previousValue)    
      console.log(cur)
      tempObject = cur
    }
    console.log(tempObject[0])
    let Y="data"
    let X = tempObject[0]
    let Z = X.split(Y).pop()
    Z=Z.replace(/\\/g, "/")
    console.log("This is the final data url:",Z)
    this.final_url=Z
    this.isImg=true;
  }

  ngOnInit(): void {
  }

  

}
