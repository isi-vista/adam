import { Component, OnInit } from '@angular/core';
import { AdamService } from 'src/app/services/adam.service';
import { FormGroup, NgForm } from '@angular/forms';
import { environment } from 'src/environments/environment';
import { TouchSequence } from 'selenium-webdriver';

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
  differences_panel: Record<string, unknown>;
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
  trainingData: string[];
  testData: string[];
  maxScenes: 10;

  selectedLearner: string;
  selectedTrain: string;
  selectedTest: string;
  selectedSceneNum: number;

  submitted = false;
  initial = 'None';
  noOutput = false;

  outputObject = {};
  differencesObject = {};
  targetImgURLs: string[];

  ngForm = FormGroup;

  constructor(private adamService: AdamService) {}

  ngOnInit(): void {
    this.adamService.getLearnerData().subscribe((data: LearnersResponse) => {
      this.learners = data.learner_types;
      this.selectedLearner = data.learner_types[0];
    });

    this.adamService
      .getTrainingData()
      .subscribe((data: TrainingCurriculumResponse) => {
        this.trainingData = data.training_curriculum;
        this.selectedTrain = data.training_curriculum[0];
      });

    this.adamService
      .getTestingData()
      .subscribe((data: TestingCurriculumResponse) => {
        this.testData = data.testing_curriculum;
        this.selectedTest = data.testing_curriculum[0];
      });

    this.selectedSceneNum = 1;
  }

  formSubmit(f: NgForm) {
    this.submitted = true;
    this.adamService
      .loadScene(
        this.selectedLearner,
        this.selectedTrain,
        this.selectedTest,
        this.selectedSceneNum
      )
      .subscribe((data: SceneResponse) => {
        if (data.message != null) {
          alert(data.message);
        }
        this.outputObject = {
          main: data.post_learning.output_language,
          scene_num: data.post_learning.scene_num,
        };
        this.targetImgURLs = data.scene_images;
        this.differencesObject = data.post_learning.differences_panel;
      });
  }

  formReset(f: NgForm) {
    f.value.selectLearner = '';
    this.submitted = false;
    this.outputObject = {};
    this.targetImgURLs = [];
  }
}
