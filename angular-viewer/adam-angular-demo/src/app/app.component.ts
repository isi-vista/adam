import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title:string = 'adam-angular-demo';
  learnerData : JSON
  preTrainingData : JSON
  trainingData : JSON
  testData : JSON

  constructor(private httpClient : HttpClient) { }

  getLearnerData(){
    this.httpClient.get("http://127.0.0.1:5000/api/learners").subscribe(data => {
      this.learnerData = data as JSON
      console.log(this.learnerData)
    })
  }
}
