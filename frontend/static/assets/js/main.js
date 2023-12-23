/*
	Dimension by HTML5 UP
	html5up.net | @ajlkn
	Free for personal and commercial use under the CCA 3.0 license (html5up.net/license)
*/

(function ($) {

	var $window = $(window),
		$body = $('body'),
		$wrapper = $('#wrapper'),
		$header = $('#header'),
		$footer = $('#footer'),
		$main = $('#main'),
		$main_articles = $main.children('article');

	// Breakpoints.
	breakpoints({
		xlarge: ['1281px', '1680px'],
		large: ['981px', '1280px'],
		medium: ['737px', '980px'],
		small: ['481px', '736px'],
		xsmall: ['361px', '480px'],
		xxsmall: [null, '360px']
	});

	// Play initial animations on page load.
	$window.on('load', function () {
		window.setTimeout(function () {
			$body.removeClass('is-preload');
		}, 100);
	});

	// Fix: Flexbox min-height bug on IE.
	if (browser.name == 'ie') {

		var flexboxFixTimeoutId;

		$window.on('resize.flexbox-fix', function () {

			clearTimeout(flexboxFixTimeoutId);

			flexboxFixTimeoutId = setTimeout(function () {

				if ($wrapper.prop('scrollHeight') > $window.height())
					$wrapper.css('height', 'auto');
				else
					$wrapper.css('height', '100vh');

			}, 250);

		}).triggerHandler('resize.flexbox-fix');

	}

	// Nav.
	var $nav = $header.children('nav'),
		$nav_li = $nav.find('li');

	// Add "middle" alignment classes if we're dealing with an even number of items.
	if ($nav_li.length % 2 == 0) {

		$nav.addClass('use-middle');
		$nav_li.eq(($nav_li.length / 2)).addClass('is-middle');

	}

	// Main.
	var delay = 325,
		locked = false;

	// Methods.
	$main._show = function (id, initial) {

		var $article = $main_articles.filter('#' + id);

		// No such article? Bail.
		if ($article.length == 0)
			return;

		// Handle lock.

		// Already locked? Speed through "show" steps w/o delays.
		if (locked || (typeof initial != 'undefined' && initial === true)) {

			// Mark as switching.
			$body.addClass('is-switching');

			// Mark as visible.
			$body.addClass('is-article-visible');

			// Deactivate all articles (just in case one's already active).
			$main_articles.removeClass('active');

			// Hide header, footer.
			$header.hide();
			$footer.hide();

			// Show main, article.
			$main.show();
			$article.show();

			// Activate article.
			$article.addClass('active');

			// Unlock.
			locked = false;

			// Unmark as switching.
			setTimeout(function () {
				$body.removeClass('is-switching');
			}, (initial ? 1000 : 0));

			return;

		}

		// Lock.
		locked = true;

		// Article already visible? Just swap articles.
		if ($body.hasClass('is-article-visible')) {

			// Deactivate current article.
			var $currentArticle = $main_articles.filter('.active');

			$currentArticle.removeClass('active');

			// Show article.
			setTimeout(function () {

				// Hide current article.
				$currentArticle.hide();

				// Show article.
				$article.show();

				// Activate article.
				setTimeout(function () {

					$article.addClass('active');

					// Window stuff.
					$window
						.scrollTop(0)
						.triggerHandler('resize.flexbox-fix');

					// Unlock.
					setTimeout(function () {
						locked = false;
					}, delay);

				}, 25);

			}, delay);

		}

		// Otherwise, handle as normal.
		else {

			// Mark as visible.
			$body
				.addClass('is-article-visible');

			// Show article.
			setTimeout(function () {

				// Hide header, footer.
				$header.hide();
				$footer.hide();

				// Show main, article.
				$main.show();
				$article.show();

				// Activate article.
				setTimeout(function () {

					$article.addClass('active');

					// Window stuff.
					$window
						.scrollTop(0)
						.triggerHandler('resize.flexbox-fix');

					// Unlock.
					setTimeout(function () {
						locked = false;
					}, delay);

				}, 25);

			}, delay);

		}

	};

	$main._hide = function (addState) {

		var $article = $main_articles.filter('.active');

		// Article not visible? Bail.
		if (!$body.hasClass('is-article-visible'))
			return;

		// Add state?
		if (typeof addState != 'undefined'
			&& addState === true)
			history.pushState(null, null, '#');

		// Handle lock.

		// Already locked? Speed through "hide" steps w/o delays.
		if (locked) {

			// Mark as switching.
			$body.addClass('is-switching');

			// Deactivate article.
			$article.removeClass('active');

			// Hide article, main.
			$article.hide();
			$main.hide();

			// Show footer, header.
			$footer.show();
			$header.show();

			// Unmark as visible.
			$body.removeClass('is-article-visible');

			// Unlock.
			locked = false;

			// Unmark as switching.
			$body.removeClass('is-switching');

			// Window stuff.
			$window
				.scrollTop(0)
				.triggerHandler('resize.flexbox-fix');

			return;

		}

		// Lock.
		locked = true;

		// Deactivate article.
		$article.removeClass('active');

		// Hide article.
		setTimeout(function () {

			// Hide article, main.
			$article.hide();
			$main.hide();

			// Show footer, header.
			$footer.show();
			$header.show();

			// Unmark as visible.
			setTimeout(function () {

				$body.removeClass('is-article-visible');

				// Window stuff.
				$window
					.scrollTop(0)
					.triggerHandler('resize.flexbox-fix');

				// Unlock.
				setTimeout(function () {
					locked = false;
				}, delay);

			}, 25);

		}, delay);


	};

	// Articles.
	$main_articles.each(function () {

		var $this = $(this);

		// Close.
		$('<div class="close">Close</div>')
			.appendTo($this)
			.on('click', function () {
				location.hash = '';
			});

		// Prevent clicks from inside article from bubbling.
		$this.on('click', function (event) {
			event.stopPropagation();
		});

	});

	// Events.
	$body.on('click', function (event) {

		// Article visible? Hide.
		if ($body.hasClass('is-article-visible'))
			$main._hide(true);

	});

	$window.on('keyup', function (event) {

		switch (event.keyCode) {

			case 27:

				// Article visible? Hide.
				if ($body.hasClass('is-article-visible'))
					$main._hide(true);

				break;

			default:
				break;

		}

	});

	$window.on('hashchange', function (event) {

		// Empty hash?
		if (location.hash == ''
			|| location.hash == '#') {

			// Prevent default.
			event.preventDefault();
			event.stopPropagation();

			// Hide.
			$main._hide();

		}

		// Otherwise, check for a matching article.
		else if ($main_articles.filter(location.hash).length > 0) {

			// Prevent default.
			event.preventDefault();
			event.stopPropagation();

			// Show article.
			$main._show(location.hash.substr(1));

		}

	});

	// Scroll restoration.
	// This prevents the page from scrolling back to the top on a hashchange.
	if ('scrollRestoration' in history)
		history.scrollRestoration = 'manual';
	else {

		var oldScrollPos = 0,
			scrollPos = 0,
			$htmlbody = $('html,body');

		$window
			.on('scroll', function () {

				oldScrollPos = scrollPos;
				scrollPos = $htmlbody.scrollTop();

			})
			.on('hashchange', function () {
				$window.scrollTop(oldScrollPos);
			});

	}

	// Initialize.

	// Hide main, articles.
	$main.hide();
	$main_articles.hide();

	// Initial article.
	if (location.hash != ''
		&& location.hash != '#')
		$window.on('load', function () {
			$main._show(location.hash.substr(1), true);
		});

})(jQuery);


function readColorImgURL(input) {
	if (input.files && input.files[0]) {

		var reader = new FileReader();


		reader.onload = function (e) {
			$('.image-color-upload-wrap').hide();

			$('.file-color-upload-image').attr('src', e.target.result);
			$('.file-color-upload-content').show();

			$('.image-color-title').html(input.files[0].name);
		};

		reader.readAsDataURL(input.files[0])

		reader.onloadend = function () {
			const xhr = new XMLHttpRequest();
			var formData = new FormData();
			formData.append('color_img', reader.result)
			xhr.open("POST", '/', true);
			xhr.send(formData);
		};
	} else {
		removeColorImgUpload();
	}
}

function readMaskImgURL(input) {
	if (input.files && input.files[0]) {

		var reader = new FileReader();


		reader.onload = function (e) {
			$('.image-mask-upload-wrap').hide();

			$('.file-mask-upload-image').attr('src', e.target.result);
			$('.file-mask-upload-content').show();

			$('.image-mask-title').html(input.files[0].name);
		};

		reader.readAsDataURL(input.files[0])

		reader.onloadend = function () {
			const xhr = new XMLHttpRequest();
			var formData = new FormData();
			formData.append('mask_img', reader.result)
			xhr.open("POST", '/', true);
			xhr.send(formData);
		};
	} else {
		removeMaskImgUpload();
	}
}



function removeColorImgUpload() {
	$('.file-color-upload-input').replaceWith($('.file-color-upload-input').clone());
	$('.file-color-upload-content').hide();
	$('.image-color-upload-wrap').show();
}
$('.image-color-upload-wrap').bind('dragover', function () {
	$('.image-color-upload-wrap').addClass('image-color-dropping');
});
$('.image-color-upload-wrap').bind('dragleave', function () {
	$('.image-color-upload-wrap').removeClass('image-color-dropping');
});

function removeMaskImgUpload() {
	$('.file-mask-upload-input').replaceWith($('.file-mask-upload-input').clone());
	$('.file-mask-upload-content').hide();
	$('.image-mask-upload-wrap').show();
}
$('.image-mask-upload-wrap').bind('dragover', function () {
	$('.image-mask-upload-wrap').addClass('image-mask-dropping');
});
$('.image-mask-upload-wrap').bind('dragleave', function () {
	$('.image-mask-upload-wrap').removeClass('image-mask-dropping');
});



function changeColorImgUpload() {
	const xhr = new XMLHttpRequest();
	var formData = new FormData();
	formData.append('convert_color_img', 'convert_color_img')
	xhr.open("POST", '/', true);
	xhr.send(formData);

	setTimeout(function () {
		document.querySelector('.file-color-upload-image').src = 'static/converted_color_img.jpg?' + new Date().getTime();
	}, 3000);
}

function changeMaskImgUpload() {
	const xhr = new XMLHttpRequest();
	var formData = new FormData();
	formData.append('convert_mask_img', 'convert_mask_img')
	xhr.open("POST", '/', true);
	xhr.send(formData);

	setTimeout(function () {
		document.querySelector('.file-mask-upload-image').src = 'static/converted_mask_img.jpg?' + new Date().getTime();
	}, 3000);
}



function uploadFiles() {
	const fileInput = document.getElementById("fileInput");
	const selectedFiles = fileInput.files;
	// Check if any files are selected
	if (selectedFiles.length === 0) {
		alert("Please select at least one file to upload.");
		return;
	}
	const formData = new FormData();
	// Append each selected file to the FormData object
	for (let i = 0; i < selectedFiles.length; i++) {
		console.log(selectedFiles[i])
		formData.append("files[]", selectedFiles[i]);
	}
	console.log([...formData])

	const xhr = new XMLHttpRequest();
	xhr.open("POST", "/", true);
	console.log(xhr.status)
	xhr.send(formData);
	setTimeout(function () {
		document.querySelector('.file-gallery-images').src = 'static/gallery_img.jpg?' + new Date().getTime();
	}, 3000);
}




