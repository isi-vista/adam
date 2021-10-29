import { Component, OnInit, Input, SimpleChange, SimpleChanges } from '@angular/core';
import { AdamService } from 'src/app/services/adam.service';
import { MainObject } from 'src/app/classes/main-object';
import { Features } from 'src/app/classes/features';
import { SubObject } from 'src/app/classes/sub-object';

@Component({
  selector: 'app-object-results',
  templateUrl: './object-results.component.html',
  styleUrls: ['./object-results.component.css']
})
export class ObjectResultsComponent implements OnInit {

  @Input() output_object;

  result_object;
  scene_number;
  sub_objects;

  result = new MainObject;
  isObject = false;

  constructor(private getResponseData: AdamService) { }

  ngOnChanges(changes: SimpleChanges){
    let tempObject;
    for (const propName in changes){
      const chng = changes[propName]
      const cur = JSON.parse(JSON.stringify(chng.currentValue))
      const prev = JSON.parse(JSON.stringify(chng.previousValue))     
      console.log(cur)
      tempObject = cur
    }
    console.log(tempObject)
    this.result.text=tempObject["main"]["text"]
    this.result.confidence=tempObject["main"]["confidence"]
    this.result.features = new Array<Features>();
    this.result.subObject = new Array<SubObject>();
    tempObject.main.features.forEach(element => {
      let feat = new Features;
      console.log(element)
      feat.name=element["name"]
      this.result.features.push(feat)
    });
    tempObject.main.sub_objects.forEach(element => {
      let subobject = new SubObject;
      subobject.confidence = element["confidence"]
      subobject.text = element["text"]
      subobject.features = new Array<Features>();
      element.features.forEach(element => {
        let feat = new Features;
        feat.name=element["name"]
        subobject.features.push(feat)
      });
      this.result.subObject.push(subobject)
    });

    console.log(this.result)
    this.isObject=true;
  }

  ngOnInit(): void {
  }
}
