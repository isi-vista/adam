import { Component, Input, OnInit } from '@angular/core';

@Component({
  selector: 'app-selectors',
  templateUrl: './selectors.component.html',
  styleUrls: ['./selectors.component.css']
})
export class SelectorsComponent implements OnInit {

  constructor() { }

  ngOnInit(): void {
  }

  @Input() function:string;
  @Input() directories:string[];

  // console.log(directories);

  log(msg: any) {
    console.log(this.directories);
}

}
