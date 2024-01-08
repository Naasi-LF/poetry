document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    form.addEventListener('submit', function(event) {
        const input = document.querySelector('input[name="start_words"]');
        if (input.value.trim() === '') {
            alert('请输入一些字符以生成诗歌。');
            event.preventDefault();
        }
    });
});