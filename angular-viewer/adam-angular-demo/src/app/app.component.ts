import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
})
export class AppComponent {
  title = 'adam-angular-demo';
  learnerData: JSON;
  preTrainingData: JSON;
  trainingData: JSON;
  testData: JSON;

  constructor(private httpClient: HttpClient) {}
}
