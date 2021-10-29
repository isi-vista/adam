import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { map } from 'rxjs/operators';
import { load } from 'js-yaml';
import { environment } from 'src/environments/environment';

@Injectable({
  providedIn: 'root',
})
export class AdamService {
  constructor(private http: HttpClient) {}

  public fetchYaml(fileName) {
    return this.http
      .get(fileName, { responseType: 'text' })
      .pipe(map((yamlString) => load(yamlString)));
  }

  public getLearnerData() {
    return this.http.get(environment.API_URL + '/api/learners');
  }

  public getTrainingData() {
    return this.http.get(environment.API_URL + '/api/training_curriculum');
  }

  public getTestingData() {
    return this.http.get(environment.API_URL + '/api/testing_curriculum');
  }

  public loadScene(
    learnerType: string,
    trainingCurriculum: string,
    testingCurriculum: string,
    sceneNum: string
  ) {
    const params = {
      learner: learnerType,
      training_curriculum: trainingCurriculum,
      testing_curriculum: testingCurriculum,
      scene_number: sceneNum,
    };
    return this.http.get(environment.API_URL + '/api/load_scene?', { params });
  }
}
