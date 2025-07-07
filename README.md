Aplicación de Predicción de Impacto de Redes Sociales
Esta aplicación web predice el impacto del uso de redes sociales en la vida de los usuarios, incluyendo el riesgo de dependencia, el rendimiento académico, la calidad del sueño y más. El backend está desarrollado con FastAPI y utiliza modelos de aprendizaje automático (Regresión Lineal, Bosques Aleatorios, Árboles de Decisión, K-Means) para generar predicciones. El frontend está construido con Vue 3, TypeScript, Tailwind CSS, e incluye animaciones, modo oscuro y visualizaciones interactivas.
Características

Formulario de Entrada: Recolecta datos del usuario (edad, género, nivel académico, país, uso de redes sociales, etc.) con secciones colapsables y tooltips.
Predicciones:
Puntaje de Riesgo de Dependencia (0-100%) con barras de progreso animadas.
Predicciones de regresión para horas de sueño, uso diario de redes sociales y puntaje de adicción.
Predicciones de clasificación para impacto académico, conflictos, uso elevado y calidad del sueño.
Agrupamiento K-Means para clasificar usuarios por comportamiento.


Mejoras de Interfaz: Animaciones (Animate.css), efectos de confeti, modo oscuro y diseño responsivo.
Visualizaciones: Gráficos de barras para modelos de regresión usando Chart.js, presentados en una interfaz de pestañas.
Seguridad de Tipos: El frontend utiliza TypeScript para un chequeo robusto de tipos.

Tecnologías Utilizadas

Backend: Python 3.8+, FastAPI, scikit-learn, pandas, joblib
Frontend: Vue 3, TypeScript, Tailwind CSS, Chart.js, Animate.css, vue3-tabs-component, canvas-confetti, Heroicons
Herramientas de Desarrollo: Vite, TypeScript, vue-tsc

Requisitos Previos

Python: 3.8 o superior
Node.js: 16 o superior
npm: 8 o superior
Git: Para clonar el repositorio
Volar: Extensión de VS Code para soporte de Vue/TypeScript

Instalación
1. Clonar el Repositorio
git clone <url-del-repositorio>
cd prediccion-impacto-redes-sociales

2. Configurar el Backend

Crear un Entorno Virtual:
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate


Instalar Dependencias de Python:
pip install -r requirements.txt


Actualizar el Mapa de Categorías:Asegúrate de que el archivo category_map.json exista en la raíz del proyecto con el siguiente contenido:
{
  "gender": ["Female", "Male", "Other"],
  "Academic_Level": ["Engineer", "Secondary", "Graduate", "University higher technician", "Preparatory", "Primary", "High_school", "Posgrado", "Secundaria", "Primaria", "Mastery", "Doctorate", "Sin estudios"],
  "Country": ["Mexico", "Usa", "Bangladesh", "Japan", "Brazil", "Uk", "Germany", "Canada", "India", "Zimbabue", "Antigua y barbuda", "Italy", "Bolivia", "Ecuador", "Ukraine", "Guatemala", "Colombia", "Peru", "Armenia", "Vietnam", "Argentina", "Chile", "Austria", "Afghanistan", "Liechtenstein", "Bahrain", "Qatar", "Malta", "España", "El salvador", "China", "Albania", "Algeria", "Belize", "Venezuela"],
  "Most_Used_Platform": ["Instagram", "Youtube", "Tiktok", "Whatsapp", "Twitter", "Facebook", "Snapchat", "Discord", "Pinterest", "Wechat", "Linkedin", "Twitter (x)", "Otra", "Vkontakte", "Line"],
  "relationship_status_single": ["Single", "Complicated", "In a relationship", "Married", "Unknown"]
}


Entrenar Modelos (si no están entrenados):
python app.py

Esto genera el directorio models/ con los modelos de aprendizaje automático entrenados.


3. Configurar el Frontend

Navegar al Directorio del Frontend (si está separado, e.g., frontend/):
cd frontend


Instalar Dependencias de Node.js:
npm install


Instalar Dependencias Adicionales:
npm install typescript vue-tsc @vue/compiler-sfc @types/node animate.css chart.js vue3-tabs-component canvas-confetti @heroicons/vue --save-dev


Configurar TypeScript:Asegúrate de que tsconfig.json exista en el directorio del frontend:
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


Actualizar main.ts:Asegúrate de que src/main.ts contenga:
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



Ejecutar la Aplicación
1. Iniciar el Backend
source venv/bin/activate  # En Windows: venv\Scripts\activate
python app.py

El servidor FastAPI se ejecuta en http://127.0.0.1:8000.
2. Iniciar el Frontend
cd frontend
npm run dev

El servidor de desarrollo de Vite se ejecuta en http://localhost:5173 (el puerto puede variar).
3. Verificación de Tipos
Ejecuta la verificación de tipos de TypeScript para el frontend:
npx vue-tsc --noEmit

4. Acceder a la Aplicación
Abre http://localhost:5173 en tu navegador. Completa el formulario con datos como:
{
  "age": 20,
  "gender": "Other",
  "Academic_Level": "Doctorate",
  "Country": "Japan",
  "avg_daily_usage_hours": 4,
  "Most_Used_Platform": "Youtube",
  "affects_academic_performance": 0,
  "sleep_hours_per_night": 7,
  "relationship_status_single": "Unknown"
}

Uso

Completar el Formulario:
Ingresa datos en las secciones colapsables (Información Personal, Uso de Redes Sociales, Estilo de Vida).
Usa los tooltips para orientación y selecciona opciones de los menús desplegables.


Enviar el Formulario:
Haz clic en "Obtener Predicciones" para enviar los datos al backend.
Observa el spinner de carga y el efecto de confeti al recibir los resultados.


Ver Resultados:
Navega por las pestañas (Riesgo de Dependencia, Predicciones Numéricas, Predicciones de Clasificación, Agrupamiento K-Means).
Expande secciones para ver detalles de los modelos y gráficos animados.


Modo Oscuro:
Usa el botón en la esquina superior derecha para alternar entre modo claro y oscuro.



Solución de Problemas

Errores de TypeScript (e.g., ts-plugin(1005)): Reinicia el servidor de TypeScript en VS Code (Ctrl+Shift+P → "TypeScript: Restart TS Server").
Errores de Backend: Verifica que category_map.json esté actualizado y que los modelos estén entrenados (models/ existe).
Gráficos No Renderizan: Asegúrate de que chart.js esté registrado en main.ts.
Errores de Conexión: Confirma que el backend esté corriendo en http://127.0.0.1:8000 y que no haya conflictos de puertos.

Contribuir

Clona el repositorio y crea una rama para tu funcionalidad:git checkout -b mi-funcionalidad


Realiza cambios y haz commit:git commit -m "Agrega mi funcionalidad"


Envía un pull request al repositorio principal.

Licencia
Este proyecto está bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.