import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { ObjectResultsComponent } from './object-results.component';

describe('ObjectResultsComponent', () => {
  let component: ObjectResultsComponent;
  let fixture: ComponentFixture<ObjectResultsComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ ObjectResultsComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(ObjectResultsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
