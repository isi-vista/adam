import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { FormGroup, NgForm } from '@angular/forms';
import { ToastrService } from 'ngx-toastr';
import { environment } from '../../../environments/environment';

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

  private apiURL = environment.API_URL;

  constructor(private http: HttpClient, public toastr: ToastrService) {}

  ngOnInit(): void {
    this.http
      .get(this.apiURL + '/api/learners')
      .toPromise()
      .then((data: LearnersResponse) => {
        this.learners = data.learner_types;
        this.selectedLearner = data.learner_types[0]
          ? data.learner_types[0]
          : this.initial;
      });

    this.http
      .get(this.apiURL + '/api/training_curriculum')
      .toPromise()
      .then((data: TrainingCurriculumResponse) => {
        this.trainingData = data.training_curriculum;
        this.selectedTrain = data.training_curriculum[0]
          ? data.training_curriculum[0]
          : this.initial;
      });

    this.http
      .get(this.apiURL + '/api/testing_curriculum')
      .toPromise()
      .then((data: TestingCurriculumResponse) => {
        this.testData = data.testing_curriculum;
        this.selectedTest = data.testing_curriculum[0]
          ? data.testing_curriculum[0]
          : this.initial;
      });

    this.selectedSceneNum = 1;
  }

  formSubmit(): number {
    if (this.selectedLearner === this.initial) {
      this.toastr.error('Invalid learner selected.');
      return 0;
    }
    if (this.selectedTest === this.initial) {
      this.toastr.error('Invalid testing curriculum selected.');
      return 0;
    }
    if (this.selectedTrain === this.initial) {
      this.toastr.error('Invalid training curriculum selected.');
      return 0;
    }
    const params = {
      learner: this.selectedLearner,
      training_curriculum: this.selectedTrain,
      testing_curriculum: this.selectedTest,
      scene_number: this.selectedSceneNum.toString(),
    };
    this.http
      .get(this.apiURL + '/api/load_scene?', { params })
      .toPromise()
      .then((data: SceneResponse) => {
        if (data.message != null) {
          this.toastr.error(data.message);
          return 0;
        }

        this.outputObject = {
          main: data.post_learning.output_language,
          scene_num: data.post_learning.scene_num,
        };
        this.targetImgURLs = data.scene_images;
        this.differencesObject = data.post_learning.differences_panel;

        return 0;
      });
    this.submitted = true;
    return 0;
  }

  formReset(f: NgForm): void {
    f.value.selectLearner = '';
    this.submitted = false;
    this.outputObject = {};
    this.targetImgURLs = [];
  }
}
