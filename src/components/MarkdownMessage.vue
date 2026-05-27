<script setup lang="ts">
import { computed } from 'vue';
import MarkdownIt from 'markdown-it';
import DOMPurify from 'dompurify';

const props = defineProps<{
  content: string;
}>();

const md = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
  typographer: true,
});

const defaultLinkOpen =
  md.renderer.rules.link_open ??
  ((tokens, idx, options, _env, self) =>
    self.renderToken(tokens, idx, options));

md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
  tokens[idx].attrSet('target', '_blank');
  tokens[idx].attrSet('rel', 'noopener noreferrer');
  return defaultLinkOpen(tokens, idx, options, env, self);
};

const renderedHtml = computed(() =>
  DOMPurify.sanitize(md.render(props.content || ''), {
    ADD_ATTR: ['target', 'rel'],
  }),
);
</script>

<template>
  <div class="prose prose-sm chat-prose max-w-none" v-html="renderedHtml" />
</template>
