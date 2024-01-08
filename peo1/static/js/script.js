document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    form.addEventListener('submit', function(event) {
        const input = document.querySelector('input[name="input_chars"]');
        if (input.value.trim() === '') {
            alert('Please enter some characters to generate the poem.');
            event.preventDefault();
        }
    });
});
