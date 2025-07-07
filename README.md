# Social Media Impact Prediction App

This project is a web application that predicts the impact of social media usage on users' lives, including dependency risk, academic performance, sleep quality, and more. The backend is built with **FastAPI** and uses machine learning models (Linear Regression, Random Forest, Decision Trees, K-Means) to generate predictions. The frontend is built with **Vue 3**, **TypeScript**, **Tailwind CSS**, and includes animations, dark mode, and interactive visualizations.

## Features
- **Form Input**: Collects user data (age, gender, academic level, country, social media usage, etc.) with collapsible sections and tooltips.
- **Predictions**:
  - Dependency Risk Score (0-100%) with animated progress bars.
  - Regression predictions for sleep hours, daily social media usage, and addiction score.
  - Classification predictions for academic impact, conflicts, high usage, and sleep quality.
  - K-Means clustering to group users by behavior.
- **UI Enhancements**: Animated transitions (Animate.css), confetti effects, dark mode toggle, and responsive design.
- **Visualizations**: Bar charts for regression models using Chart.js, displayed in a tabbed interface.
- **Type Safety**: Frontend uses TypeScript for robust type checking.

## Tech Stack
- **Backend**: Python 3.8+, FastAPI, scikit-learn, pandas, joblib
- **Frontend**: Vue 3, TypeScript, Tailwind CSS, Chart.js, Animate.css, vue3-tabs-component, canvas-confetti, Heroicons
- **Development Tools**: Vite, TypeScript, vue-tsc

## Prerequisites
- **Python**: 3.8 or higher
- **Node.js**: 16 or higher
- **npm**: 8 or higher
- **Git**: For cloning the repository
- **Volar**: VS Code extension for Vue/TypeScript support

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd social-media-impact-prediction
```

### 2. Set Up the Backend
1. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Update Category Map**:
   Ensure `category_map.json` exists in the project root with the following content:
   ```json
   {
     "gender": ["Female", "Male", "Other"],
     "Academic_Level": ["Engineer", "Secondary", "Graduate", "University higher technician", "Preparatory", "Primary", "High_school", "Posgrado", "Secundaria", "Primaria", "Mastery", "Doctorate", "Sin estudios"],
     "Country": ["Mexico", "Usa", "Bangladesh", "Japan", "Brazil", "Uk", "Germany", "Canada", "India", "Zimbabue", "Antigua y barbuda", "Italy", "Bolivia", "Ecuador", "Ukraine", "Guatemala", "Colombia", "Peru", "Armenia", "Vietnam", "Argentina", "Chile", "Austria", "Afghanistan", "Liechtenstein", "Bahrain", "Qatar", "Malta", "España", "El salvador", "China", "Albania", "Algeria", "Belize", "Venezuela"],
     "Most_Used_Platform": ["Instagram", "Youtube", "Tiktok", "Whatsapp", "Twitter", "Facebook", "Snapchat", "Discord", "Pinterest", "Wechat", "Linkedin", "Twitter (x)", "Otra", "Vkontakte", "Line"],
     "relationship_status_single": ["Single", "Complicated", "In a relationship", "Married", "Unknown"]
   }
   ```

4. **Train Models** (if not already trained):
   ```bash
   python app.py
   ```
   This generates the `models/` directory with trained machine learning models.

### 3. Set Up the Frontend
1. **Navigate to the Frontend Directory** (if separate, e.g., `frontend/`):
   ```bash
   cd frontend
   ```

2. **Install Node.js Dependencies**:
   ```bash
   npm install
   ```

3. **Install Additional Dependencies**:
   ```bash
   npm install typescript vue-tsc @vue/compiler-sfc @types/node animate.css chart.js vue3-tabs-component canvas-confetti @heroicons/vue --save-dev
   ```

4. **Configure TypeScript**:
   Ensure `tsconfig.json` exists in the frontend directory:
   ```json
   {
     "compilerOptions": {
       "target": "esnext",
       "module": "esnext",
       "moduleResolution": "node",
       "strict": true,
       "jsx": "preserve",
       "esModuleInterop": true,
       "skipLibCheck": true,
       "forceConsistentCasingInFileNames": true,
       "resolveJsonModule": true,
       "baseUrl": ".",
       "paths": {
         "@/*": ["src/*"]
       },
       "types": ["node", "vite/client"]
     },
     "include": ["src/**/*", "src/**/*.vue"],
     "exclude": ["node_modules"]
   }
   ```

5. **Update `main.ts`**:
   Ensure `src/main.ts` contains:
   ```typescript
   import { createApp } from 'vue';
   import App from './App.vue';
   import { Tabs, Tab } from 'vue3-tabs-component';
   import ChartJs from 'chart.js/auto';

   const app = createApp(App);
   app.component('Tabs', Tabs);
   app.component('Tab', Tab);
   app.component('chartjs', {
     props: { config: Object },
     mounted() {
       new ChartJs(this.$el, this.config);
     },
     template: '<canvas></canvas>',
   });
   app.mount('#app');
   ```

## Running the Application

### 1. Start the Backend
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python app.py
```
The FastAPI server runs on `http://127.0.0.1:8000`.

### 2. Start the Frontend
```bash
cd frontend
npm run dev
```
The Vite development server runs on `http://localhost:5173` (port may vary).

### 3. Type Checking
Run TypeScript type checking for the frontend:
```bash
npx vue-tsc --noEmit
```

### 4. Access the App
Open `http://localhost:5173` in your browser. Fill out the form with data such as:
```json
{
  "age": 20,
  "gender": "Other",
  "Academic_Level": "Doctorate",
  "Country": "Japan",
  "avg_daily_usage_hours": 4,
  "Most_Used_Platform": "Youtube",
  "affects_academic_performance": 0,
  "sleep
}
```
