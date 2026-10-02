import { bindThis } from '@/decorators.js';
import Module from '@/module.js';
import serifs from '@/serifs.js';
import { genMaze } from './gen-maze.js';
import { renderMaze } from './render-maze.js';
import Message from '@/message.js';

export default class extends Module {
	public readonly name = 'maze';

	private readonly difficulties = [
		{ key: 'veryEasy', label: '激かんたん' },
		{ key: 'easy', label: 'かんたん' },
		{ key: 'normal', label: 'ふつう' },
		{ key: 'hard', label: 'むずかしい' },
		{ key: 'veryHard', label: '激ムズ' },
		{ key: 'ai', label: '藍本気' },
	];

	private readonly postSlots = [
		{ hour: 7, label: '朝' },
		{ hour: 13, label: '昼' },
		{ hour: 19, label: '晩' },
	];

	@bindThis
	public install() {
		this.post();
		setInterval(this.post, 1000 * 60 * 3);

		return {
			mentionHook: this.mentionHook
		};
	}

	@bindThis
	private async post() {
		// サーバーのシステムタイムゾーンに関係なく、常にJST(UTC+9)基準で判定する
		const jst = new Date(Date.now() + 9 * 60 * 60 * 1000);
		const slot = this.postSlots.find(s => s.hour === jst.getUTCHours());
		if (slot == null) return;
		const date = `${jst.getUTCFullYear()}-${jst.getUTCMonth()}-${jst.getUTCDate()}`;
		const key = `${date}-${slot.hour}`;
		const data = this.getData();
		if (data.lastPosted == key) return;
		data.lastPosted = key;
		this.setData(data);

		this.log('Time to maze');
		const difficulty = this.difficulties[Math.floor(Math.random() * this.difficulties.length)];
		const file = await this.genMazeFile(key, difficulty.key);

		this.log('Posting...');
		const text = serifs.maze.post(slot.label, difficulty.label);
		this.ai.post({
			...(difficulty.key === 'ai' ? { cw: serifs.maze.trypophobiaCw(text) } : { text }),
			fileIds: [file.id]
		});
	}

	@bindThis
	private async genMazeFile(seed, size?): Promise<any> {
		this.log('Maze generating...');
		const maze = genMaze(seed, size);

		this.log('Maze rendering...');
		const data = renderMaze(seed, maze);

		this.log('Image uploading...');
		const file = await this.ai.upload(data, {
			filename: 'maze.png',
			contentType: 'image/png'
		});

		return file;
	}

	@bindThis
	private async mentionHook(msg: Message) {
		if (msg.includes(['迷路'])) {
			let size: string | null = null;
			if (msg.includes(['接待'])) size = 'veryEasy';
			if (msg.includes(['簡単', 'かんたん', '易しい', 'やさしい', '小さい', 'ちいさい'])) size = 'easy';
			if (msg.includes(['難しい', 'むずかしい', '複雑な', '大きい', 'おおきい'])) size = 'hard';
			if (msg.includes(['死', '鬼', '地獄'])) size = 'veryHard';
			if (msg.includes(['藍']) && msg.includes(['本気'])) size = 'ai';
			this.log('Maze requested');
			setTimeout(async () => {
				const file = await this.genMazeFile(Date.now(), size);
				this.log('Replying...');
				if (size === 'ai') {
					msg.reply(null, { file, cw: serifs.maze.trypophobiaCw(serifs.maze.foryou) });
				} else {
					msg.reply(serifs.maze.foryou, { file });
				}
			}, 3000);
			return {
				reaction: 'like'
			};
		} else {
			return false;
		}
	}
}
