import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { SelectorParentComponent } from './selector-parent.component';

describe('SelectorParentComponent', () => {
  let component: SelectorParentComponent;
  let fixture: ComponentFixture<SelectorParentComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ SelectorParentComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(SelectorParentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
