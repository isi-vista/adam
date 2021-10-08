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
    this.fetchYaml('../../../../../data/learners/integrated_subset/experiments/objects_one/test_curriculums/objects_one_instance/test_curriculums/object_test_curriculum_one/situation_1/post_decode.yaml')
   }

  public fetchYaml(fileName) {
    return this.http.get(fileName,{responseType: 'text'}).pipe(
      map(yamlString => load(yamlString))
    )
  }
}
