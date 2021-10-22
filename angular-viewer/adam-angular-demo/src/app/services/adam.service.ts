import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { fromEventPattern, Observable, pipe, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';
// import {parse} from 'yamljs';
import {map} from 'rxjs/operators';
import {load,loadAll} from 'js-yaml'
import { environment } from 'src/environments/environment';


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
    return this.http.get(environment.API_URL+"/api/learners")
  }

  public getTrainingData(){
    return this.http.get(environment.API_URL+"/api/training_curriculum")
  }

  public getTestingData(){
    return this.http.get(environment.API_URL+"/api/testing_curriculum")
  }

  public loadScene(url:URL){
    let learner:string = url.searchParams.get("learner")
    let training_curriculum:string = url.searchParams.get("training_curriculum")
    let testing_curriculum:string = url.searchParams.get("testing_curriculum")
    let scene_number:string = url.searchParams.get("scene_number")
    return this.http.get(environment.API_URL+"/api/load_scene?"+"learner="+learner+"&training_curriculum="+training_curriculum+"&testing_curriculum="+testing_curriculum+"&scene_number="+scene_number)
  }
}
