const dropbox = document.querySelector('.dropbox');
const input = document.getElementById('new-pdf-input');
const fileName = document.getElementById('new-pdf-name');

if (dropbox && input && fileName) {
  const updateName = () => {
    if (input.files && input.files[0]) {
      fileName.textContent = input.files[0].name;
    } else {
      fileName.textContent = 'No file selected';
    }
  };

  dropbox.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropbox.classList.add('dragover');
  });

  dropbox.addEventListener('dragleave', () => {
    dropbox.classList.remove('dragover');
  });

  dropbox.addEventListener('drop', (e) => {
    e.preventDefault();
    dropbox.classList.remove('dragover');
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      updateName();
    }
  });

  dropbox.addEventListener('click', () => input.click());
  input.addEventListener('change', updateName);
}
