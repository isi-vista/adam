import { Component, OnInit } from '@angular/core';
import { SelectorsComponent } from '../selectors/selectors.component';
import { ButtonComponent } from '../button/button.component';

@Component({
  selector: 'app-selector-parent',
  templateUrl: './selector-parent.component.html',
  styleUrls: ['./selector-parent.component.css']
})
export class SelectorParentComponent implements OnInit {

  learners: string[] = ['integrated_pursuit_object_only','integrated_subset']
  pretraining_data: string[] = ['objects_one','objects_two','objects_three']
  training_data: string[] = ['1','2','3']
  test_data: string[] = ['objects_one_instance','objects_two_instance','objects_three_instance']

  constructor() { }

  ngOnInit(): void {
  }

  onButtonClick(){
    console.log("A button has been clicked")
  }

}
