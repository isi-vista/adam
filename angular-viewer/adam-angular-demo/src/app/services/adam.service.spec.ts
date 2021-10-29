import { TestBed } from '@angular/core/testing';

import { AdamService } from './adam.service';

describe('AdamService', () => {
  let service: AdamService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(AdamService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
