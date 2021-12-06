import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';

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

  constructor() {}

  reset(): void {
    this.sceneImageArray = [];
    this.sceneStrokeArray = [];
    this.sceneStrokeGraphArray = [];
  }

  ngOnChanges(changes: SimpleChanges): void {
    this.reset();

    for (const path_id in changes.scene_images.currentValue) {
      this.sceneImageArray.push(
        this.suffix +
          changes.scene_images.currentValue[path_id].replace(/\\/g, '/')
      );
    }
    for (const path_id in changes.stroke_images.currentValue) {
      this.sceneStrokeArray.push(
        this.suffix +
          changes.stroke_images.currentValue[path_id].replace(/\\/g, '/')
      );
    }
    for (const path_id in changes.stroke_graph_images.currentValue) {
      this.sceneStrokeGraphArray.push(
        this.suffix +
          changes.stroke_graph_images.currentValue[path_id].replace(/\\/g, '/')
      );
    }

    this.isImg = true;
  }

  onClickCarousel(selection: string): void {
    this.carouselSelection = selection;
  }
}
