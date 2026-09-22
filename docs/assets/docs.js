'use strict';
const field = document.querySelector('#search');
if (field) {
  const cards = [...document.querySelectorAll('.card')];
  const count = document.querySelector('#result-count');
  const empty = document.querySelector('#empty');
  field.addEventListener('input', () => {
    const terms = field.value.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
    let visible = 0;
    for (const card of cards) {
      card.hidden = !terms.every(term => card.dataset.search.includes(term));
      if (!card.hidden) visible++;
    }
    count.textContent = `${visible} of ${cards.length} chapters`;
    empty.hidden = visible !== 0;
  });
}
document.querySelectorAll('[data-print]').forEach(button => {
  button.hidden = false;
  button.addEventListener('click', () => window.print());
});
