import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { AppComponent } from './app.component';
import { HeaderComponent } from './components/header/header.component';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { SelectorsComponent } from './components/selectors/selectors.component';
import { SelectorParentComponent } from './components/selector-parent/selector-parent.component';
import { ButtonComponent } from './components/button/button.component';
import { ImageOutputComponent } from './components/image-output/image-output.component';
import { ObjectResultsComponent } from './components/object-results/object-results.component';
import { HttpClientModule } from '@angular/common/http';

@NgModule({
  declarations: [
    AppComponent,
    HeaderComponent,
    SelectorsComponent,
    SelectorParentComponent,
    ButtonComponent,
    ImageOutputComponent,
    ObjectResultsComponent,
  ],
  imports: [FormsModule, BrowserModule, NgbModule, HttpClientModule],
  providers: [],
  bootstrap: [AppComponent],
})
export class AppModule {}
