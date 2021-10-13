import { Component, OnInit } from '@angular/core';
import { SelectorsComponent } from '../selectors/selectors.component';
import { ButtonComponent } from '../button/button.component';
import { HttpClient } from '@angular/common/http';
import { AdamService } from 'src/app/services/adam.service';

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

}
