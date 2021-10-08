import { Component, OnInit } from '@angular/core';
import { AdamService } from 'src/app/services/adam.service';

@Component({
  selector: 'app-object-results',
  templateUrl: './object-results.component.html',
  styleUrls: ['./object-results.component.css']
})
export class ObjectResultsComponent implements OnInit {

  result_object;
  scene_number;
  sub_objects;

  constructor(private getResponseData: AdamService) { }

  ngOnInit(): void {
    this.getResponseData.fetchYaml("../../../assets/learners/integrated_subset/experiments/objects_one/test_curriculums/objects_one_instance/test_curriculums/object_test_curriculum_one/situation_1/post_decode.yaml").subscribe(response => {
      console.log(response)
      this.result_object=response.output_language[0];
      this.sub_objects=response.output_language[0].sub_objects[0];
      this.scene_number=response.scene_num;
      console.log(this.result_object);
      console.log(this.scene_number);
      console.log(this.sub_objects);
    });
  }

}
