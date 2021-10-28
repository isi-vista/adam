import { Component, OnInit } from '@angular/core';
import { AdamService } from 'src/app/services/adam.service';
import { NgForm } from '@angular/forms';
import { environment } from 'src/environments/environment';

@Component({
  selector: 'app-selector-parent',
  templateUrl: './selector-parent.component.html',
  styleUrls: ['./selector-parent.component.css'],
})
export class SelectorParentComponent implements OnInit {

  learners: string[]
  pretraining_data: string[]
  training_data: string[]
  test_data: string[]
  learnerData : JSON
  preTrainingData : JSON
  trainingData : JSON
  testData : JSON
  selectedLevel: string='';
  selectedLearner: string='';
  selectedPretrain: string="objects_one"
  selectedTrain: string="objects_one"

  outputImage:string="";
  outputObject={};
  targetImgURLs:string[];

  constructor(private getResponseData : AdamService) { }

  ngOnInit(): void {
    this.getResponseData.getLearnerData().subscribe(data => {
      this.learnerData = data as JSON
      this.learners = this.learnerData["learner_types"]
      console.log(this.learnerData)
      console.log(this.learners)
    })

    this.getResponseData.getTrainingData().subscribe(data => {
      this.preTrainingData = data as JSON
      this.trainingData = data as JSON
      this.pretraining_data = this.preTrainingData["training_curriculum"]
      this.training_data = this.trainingData["training_curriculum"]
      console.log(this.preTrainingData)
    })

    this.getResponseData.getTestingData().subscribe(data => {
      this.testData = data as JSON
      this.test_data = this.testData["testing_curriculum"]
      console.log(this.testData)
    })
  }

  onButtonClick(){
    console.log("A button has been clicked")
  }

  learner_selected(event: any){
    this.selectedLearner=event.target.value;
  }

  pretraining_selected(event: any){
    this.selectedPretrain=event.target.value;
  }

  training_selected(event: any){
    this.selectedTrain=event.target.value;
  }

  selected(event: any){
    this.selectedLevel = event.target.value;
    this.getResponseData.loadSceneTwo(this.selectedLearner,this.selectedTrain,this.selectedLevel,"1").subscribe(data => {
      console.log(data)
      this.outputImage=data["scene_images"][0]
      let result = {}
      result["main"]=data["post_learning"]["output_language"][0]
      let sub_objects=data["post_learning"]["output_language"][0]["sub_objects"][0]
      result["sub_objects"]=sub_objects
      let scene_num = data["post_learning"]["scene_num"]
      result["scene_num"]=scene_num
      this.outputObject=result
      this.targetImgURLs=data["scene_images"]
      console.log("Image url ",this.outputImage)
      console.log("Scene Number:",scene_num);
      console.log("Sub Objects:",sub_objects);
      console.log("Main output object: ",this.outputObject)

    })
  }

  formSubmit(form: NgForm){
    const learner_val = form.controls['selectLearner'].value;

  }

}
