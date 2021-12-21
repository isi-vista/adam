import { Component, EventEmitter, Output } from '@angular/core';

@Component({
  selector: 'app-button',
  templateUrl: './button.component.html',
  styleUrls: ['./button.component.css'],
})
export class ButtonComponent {
  @Output() btnClick = new EventEmitter();

  constructor() {}

  OnClick(): void {
    this.btnClick.emit();
  }
}
