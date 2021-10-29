import { Component, OnInit } from '@angular/core';
import { AdamService } from 'src/app/services/adam.service';
import { NgForm } from '@angular/forms';
import { environment } from 'src/environments/environment';

export interface LearnersResponse {
  learner_types: string[];
}

export interface TrainingCurriculumResponse {
  training_curriculum: string[];
}

export interface TestingCurriculumResponse {
  testing_curriculum: string[];
}

export interface FeatureResponse {
  name: string;
}

export interface LanguageResponse {
  text: string;
  confidence: number;
  features: FeatureResponse[];
  sub_objects: LanguageResponse[];
}

export interface DecodeResponse {
  scene_num: number;
  output_language: LanguageResponse[];
}

export interface SceneResponse {
  learner: string;
  train_curriculum: string;
  test_curriculum: string;
  scene_number: string;
  scene_images: string[];
  post_learning: DecodeResponse;
  pre_learning: DecodeResponse;
  message?: string;
}

@Component({
  selector: 'app-selector-parent',
  templateUrl: './selector-parent.component.html',
  styleUrls: ['./selector-parent.component.css'],
})
export class SelectorParentComponent implements OnInit {
  learners: string[];
  pretrainingData: string[];
  trainingData: string[];
  testData: string[];
  selectedLevel = '';
  selectedLearner = '';
  selectedPretrain = '';
  selectedTrain = '';

  outputImage = '';
  outputObject = {};
  targetImgURLs: string[];

  constructor(private getResponseData: AdamService) {}

  ngOnInit(): void {
    this.getResponseData
      .getLearnerData()
      .subscribe((data: LearnersResponse) => {
        this.learners = data.learner_types;
        this.selectedLearner = data.learner_types[0];
        console.log(this.learners);
      });

    this.getResponseData
      .getTrainingData()
      .subscribe((data: TrainingCurriculumResponse) => {
        this.pretrainingData = data.training_curriculum;
        this.trainingData = data.training_curriculum;
        this.selectedTrain = data.training_curriculum[0];
        console.log(this.trainingData);
      });

    this.getResponseData
      .getTestingData()
      .subscribe((data: TestingCurriculumResponse) => {
        this.testData = data.testing_curriculum;
        console.log(this.testData);
      });
  }

  onButtonClick() {
    console.log('A button has been clicked');
  }

  learner_selected(event: any) {
    this.selectedLearner = event.target.value;
  }

  pretraining_selected(event: any) {
    this.selectedPretrain = event.target.value;
  }

  training_selected(event: any) {
    this.selectedTrain = event.target.value;
  }

  selected(event: any) {
    this.selectedLevel = event.target.value;
    this.getResponseData
      .loadScene(
        this.selectedLearner,
        this.selectedTrain,
        this.selectedLevel,
        '1'
      )
      .subscribe((data: SceneResponse) => {
        console.log(data);
        this.outputImage = data.scene_images[0];
        this.outputObject = {
          main: data.post_learning.output_language[0],
          sub_objects: data.post_learning.output_language[0].sub_objects[0],
          scene_num: data.post_learning.scene_num,
        };
        this.targetImgURLs = data.scene_images;
        console.log('Image url ', this.outputImage);
        console.log('Main output object: ', this.outputObject);
      });
  }

  formSubmit(form: NgForm) {
    const learnerVal = form.controls.selectedLearner.value;
  }
}
