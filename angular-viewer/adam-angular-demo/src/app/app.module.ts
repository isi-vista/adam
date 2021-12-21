import { HttpClientModule } from '@angular/common/http';
import { NgModule } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { ToastrModule } from 'ngx-toastr';
import { AppComponent } from './app.component';
import { ButtonComponent } from './components/button/button.component';
import { HeaderComponent } from './components/header/header.component';
import { ImageOutputComponent } from './components/image-output/image-output.component';
import { ObjectResultsComponent } from './components/object-results/object-results.component';
import { PanelViewerComponent } from './components/panel-viewer/panel-viewer.component';
import { SelectorParentComponent } from './components/selector-parent/selector-parent.component';

@NgModule({
  declarations: [
    AppComponent,
    HeaderComponent,
    SelectorParentComponent,
    ButtonComponent,
    ImageOutputComponent,
    ObjectResultsComponent,
    PanelViewerComponent,
  ],
  imports: [
    FormsModule,
    BrowserModule,
    NgbModule,
    HttpClientModule,
    ToastrModule.forRoot(),
    BrowserAnimationsModule,
  ],
  providers: [],
  bootstrap: [AppComponent],
})
export class AppModule {}
