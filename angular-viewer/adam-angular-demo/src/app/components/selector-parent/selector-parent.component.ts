import { Component, OnInit } from '@angular/core';
import { SelectorsComponent } from '../selectors/selectors.component';
import { ButtonComponent } from '../button/button.component';
import { HttpClient } from '@angular/common/http';
import { AdamService } from 'src/app/services/adam.service';
import { Router } from '@angular/router';
import { NgForm } from '@angular/forms';

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
    let url = new URL("http://127.0.0.1:5000/api/load_scene");
    this.selectedLevel = event.target.value;
    url.searchParams.set("learner",this.selectedLearner)
    url.searchParams.set("training_curriculum",this.selectedTrain)
    url.searchParams.set("testing_curriculum",this.selectedLevel)
    url.searchParams.set("scene_number","1")
    console.log(this.selectedLevel)
    console.log(url)
    this.getResponseData.loadScene(url).subscribe(data => {
      console.log(data)
    })
  }

  formSubmit(form: NgForm){
    const learner_val = form.controls['selectLearner'].value;
    console.log(learner_val)
  }

}
