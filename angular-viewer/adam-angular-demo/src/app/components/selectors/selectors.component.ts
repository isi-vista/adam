import { Component, Input, OnInit } from '@angular/core';

@Component({
  selector: 'app-selectors',
  templateUrl: './selectors.component.html',
  styleUrls: ['./selectors.component.css'],
})
export class SelectorsComponent implements OnInit {
  @Input() function: string;
  @Input() directories: string[];

  selectedTest;
  selectedTrain;
  selectedPretrain;
  selectedLearner;

  constructor() {}

  ngOnInit(): void {}

  log(msg: any) {
    console.log(this.directories);
  }

  selected() {
    console.log(this.selectedTest);
  }
}
