import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { fromEventPattern, Observable, pipe, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';
// import {parse} from 'yamljs';
import {map} from 'rxjs/operators';
import {load,loadAll} from 'js-yaml'


@Injectable({
  providedIn: 'root'
})
export class AdamService {

  // getResults() {
  //   return "results";
  // }
  constructor(private http: HttpClient) {
    
   }

  public fetchYaml(fileName) {
    return this.http.get(fileName,{responseType: 'text'}).pipe(
      map(yamlString => load(yamlString))
    )
  }

  public getLearnerData(){
    return this.http.get("http://127.0.0.1:5000/api/learners")
  }

  public getTrainingData(){
    return this.http.get("http://127.0.0.1:5000/api/training_curriculum")
  }

  public getTestingData(){
    return this.http.get("http://127.0.0.1:5000/api/testing_curriculum")
  }
}
