import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { PanelViewerComponent } from './panel-viewer.component';

describe('PanelViewerComponent', () => {
  let component: PanelViewerComponent;
  let fixture: ComponentFixture<PanelViewerComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [PanelViewerComponent],
    }).compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(PanelViewerComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
