import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { ImageOutputComponent } from './image-output.component';

describe('ImageOutputComponent', () => {
  let component: ImageOutputComponent;
  let fixture: ComponentFixture<ImageOutputComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ImageOutputComponent],
    }).compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(ImageOutputComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
