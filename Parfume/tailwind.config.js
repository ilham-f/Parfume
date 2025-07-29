/** @type {import('tailwindcss').Config} */

module.exports = {
  content: ['./templates/**/*.html', './static/css/**/*.css'],
  theme: {
    extend: {
      screens: {
        '2xs': '475px',
        'xs': '768px',
        'sm': '1024px',
        'md': '1280px',
        'lg': '1536px',
      },
    },
  },
  plugins: [],
}