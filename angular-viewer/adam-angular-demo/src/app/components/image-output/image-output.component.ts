import { HttpClient } from '@angular/common/http';
import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { ToastrService } from 'ngx-toastr';
import { environment } from '../../../environments/environment';

export interface ImageResponse {
  image_data: string;
  message?: string;
}

@Component({
  selector: 'app-image-output',
  templateUrl: './image-output.component.html',
  styleUrls: ['./image-output.component.css'],
})
export class ImageOutputComponent implements OnChanges {
  @Input() scene_images: string[] = [];
  @Input() stroke_images: string[] = [];
  @Input() stroke_graph_images: string[] = [];

  isImg = false;
  carouselSelection = 'rgb';

  sceneImageArray = [];
  sceneStrokeArray = [];
  sceneStrokeGraphArray = [];
  suffix = '../../../assets';

  private apiURL = environment.API_URL;

  constructor(private http: HttpClient, public toastr: ToastrService) {}

  reset(): void {
    this.sceneImageArray = [];
    this.sceneStrokeArray = [];
    this.sceneStrokeGraphArray = [];
  }

  ngOnChanges(changes: SimpleChanges): void {
    this.reset();
    this.sceneImageArray = changes.scene_images.currentValue;
    this.sceneStrokeArray = changes.stroke_images.currentValue;
    this.sceneStrokeGraphArray = changes.stroke_graph_images.currentValue;
    this.isImg = true;
  }

  onClickCarousel(selection: string): void {
    this.carouselSelection = selection;
  }
}
